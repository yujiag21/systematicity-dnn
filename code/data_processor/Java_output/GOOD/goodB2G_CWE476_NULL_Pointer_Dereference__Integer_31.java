    private void goodB2G() throws Throwable
    {
        Integer dataCopy;
        {
            Integer data;

            /* POTENTIAL FLAW: data is null */
            data = null;

            dataCopy = data;
        }
        {
            Integer data = dataCopy;

            /* FIX: validate that data is non-null */
            if (data != null)
            {
                IO.writeLine("" + data.toString());
            }
            else
            {
                IO.writeLine("data is null");
            }

        }
    }
