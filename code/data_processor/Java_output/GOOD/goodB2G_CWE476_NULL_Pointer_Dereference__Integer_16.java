    private void goodB2G() throws Throwable
    {
        Integer data;

        while (true)
        {
            /* POTENTIAL FLAW: data is null */
            data = null;
            break;
        }

        while (true)
        {
            /* FIX: validate that data is non-null */
            if (data != null)
            {
                IO.writeLine("" + data.toString());
            }
            else
            {
                IO.writeLine("data is null");
            }
            break;
        }
    }
