    private void goodG2B() throws Throwable
    {
        Integer dataCopy;
        {
            Integer data;

            /* FIX: hardcode data to non-null */
            data = Integer.valueOf(5);

            dataCopy = data;
        }
        {
            Integer data = dataCopy;

            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.toString());

        }
    }
